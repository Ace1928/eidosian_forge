from collections import namedtuple
from fontTools.cffLib import (
from io import BytesIO
from fontTools.cffLib.specializer import specializeCommands, commandsToProgram
from fontTools.ttLib import newTable
from fontTools import varLib
from fontTools.varLib.models import allEqual
from fontTools.misc.roundTools import roundFunc
from fontTools.misc.psCharStrings import T2CharString, T2OutlineExtractor
from fontTools.pens.t2CharStringPen import T2CharStringPen
from functools import partial
from .errors import (
def merge_PrivateDicts(top_dicts, vsindex_dict, var_model, fd_map):
    """
    I step through the FontDicts in the FDArray of the varfont TopDict.
    For each varfont FontDict:

    * step through each key in FontDict.Private.
    * For each key, step through each relevant source font Private dict, and
            build a list of values to blend.

    The 'relevant' source fonts are selected by first getting the right
    submodel using ``vsindex_dict[vsindex]``. The indices of the
    ``subModel.locations`` are mapped to source font list indices by
    assuming the latter order is the same as the order of the
    ``var_model.locations``. I can then get the index of each subModel
    location in the list of ``var_model.locations``.
    """
    topDict = top_dicts[0]
    region_top_dicts = top_dicts[1:]
    if hasattr(region_top_dicts[0], 'FDArray'):
        regionFDArrays = [fdTopDict.FDArray for fdTopDict in region_top_dicts]
    else:
        regionFDArrays = [[fdTopDict] for fdTopDict in region_top_dicts]
    for fd_index, font_dict in enumerate(topDict.FDArray):
        private_dict = font_dict.Private
        vsindex = getattr(private_dict, 'vsindex', 0)
        sub_model, _ = vsindex_dict[vsindex]
        master_indices = []
        for loc in sub_model.locations[1:]:
            i = var_model.locations.index(loc) - 1
            master_indices.append(i)
        pds = [private_dict]
        last_pd = private_dict
        for ri in master_indices:
            pd = get_private(regionFDArrays, fd_index, ri, fd_map)
            if pd is None:
                pd = last_pd
            else:
                last_pd = pd
            pds.append(pd)
        num_masters = len(pds)
        for key, value in private_dict.rawDict.items():
            dataList = []
            if key not in pd_blend_fields:
                continue
            if isinstance(value, list):
                try:
                    values = [pd.rawDict[key] for pd in pds]
                except KeyError:
                    print('Warning: {key} in default font Private dict is missing from another font, and was discarded.'.format(key=key))
                    continue
                try:
                    values = zip(*values)
                except IndexError:
                    raise VarLibCFFDictMergeError(key, value, values)
                '\n\t\t\t\tRow 0 contains the first  value from each master.\n\t\t\t\tConvert each row from absolute values to relative\n\t\t\t\tvalues from the previous row.\n\t\t\t\te.g for three masters,\ta list of values was:\n\t\t\t\tmaster 0 OtherBlues = [-217,-205]\n\t\t\t\tmaster 1 OtherBlues = [-234,-222]\n\t\t\t\tmaster 1 OtherBlues = [-188,-176]\n\t\t\t\tThe call to zip() converts this to:\n\t\t\t\t[(-217, -234, -188), (-205, -222, -176)]\n\t\t\t\tand is converted finally to:\n\t\t\t\tOtherBlues = [[-217, 17.0, 46.0], [-205, 0.0, 0.0]]\n\t\t\t\t'
                prev_val_list = [0] * num_masters
                any_points_differ = False
                for val_list in values:
                    rel_list = [val - prev_val_list[i] for i, val in enumerate(val_list)]
                    if not any_points_differ and (not allEqual(rel_list)):
                        any_points_differ = True
                    prev_val_list = val_list
                    deltas = sub_model.getDeltas(rel_list)
                    deltas[0] = val_list[0]
                    dataList.append(deltas)
                if not any_points_differ:
                    dataList = [data[0] for data in dataList]
            else:
                values = [pd.rawDict[key] for pd in pds]
                if not allEqual(values):
                    dataList = sub_model.getDeltas(values)
                else:
                    dataList = values[0]
            if isinstance(dataList, list):
                for i, item in enumerate(dataList):
                    if isinstance(item, list):
                        for j, jtem in enumerate(item):
                            dataList[i][j] = conv_to_int(jtem)
                    else:
                        dataList[i] = conv_to_int(item)
            else:
                dataList = conv_to_int(dataList)
            private_dict.rawDict[key] = dataList