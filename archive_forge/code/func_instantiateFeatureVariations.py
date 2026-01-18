from fontTools.ttLib.tables import otTables as ot
from copy import deepcopy
import logging
def instantiateFeatureVariations(varfont, axisLimits):
    for tableTag in ('GPOS', 'GSUB'):
        if tableTag not in varfont or not getattr(varfont[tableTag].table, 'FeatureVariations', None):
            continue
        log.info('Instantiating FeatureVariations of %s table', tableTag)
        _instantiateFeatureVariations(varfont[tableTag].table, varfont['fvar'].axes, axisLimits)
        varfont[tableTag].prune_lookups()