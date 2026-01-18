from collections import namedtuple, OrderedDict
import os
from fontTools.misc.fixedTools import fixedToFloat
from fontTools.misc.roundTools import otRound
from fontTools import ttLib
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables.otBase import (
from fontTools.ttLib.tables import otBase
from fontTools.feaLib.ast import STATNameStatement
from fontTools.otlLib.optimize.gpos import (
from fontTools.otlLib.error import OpenTypeLibError
from functools import reduce
import logging
import copy
def buildStatTable(ttFont, axes, locations=None, elidedFallbackName=2, windowsNames=True, macNames=True):
    """Add a 'STAT' table to 'ttFont'.

    'axes' is a list of dictionaries describing axes and their
    values.

    Example::

        axes = [
            dict(
                tag="wght",
                name="Weight",
                ordering=0,  # optional
                values=[
                    dict(value=100, name='Thin'),
                    dict(value=300, name='Light'),
                    dict(value=400, name='Regular', flags=0x2),
                    dict(value=900, name='Black'),
                ],
            )
        ]

    Each axis dict must have 'tag' and 'name' items. 'tag' maps
    to the 'AxisTag' field. 'name' can be a name ID (int), a string,
    or a dictionary containing multilingual names (see the
    addMultilingualName() name table method), and will translate to
    the AxisNameID field.

    An axis dict may contain an 'ordering' item that maps to the
    AxisOrdering field. If omitted, the order of the axes list is
    used to calculate AxisOrdering fields.

    The axis dict may contain a 'values' item, which is a list of
    dictionaries describing AxisValue records belonging to this axis.

    Each value dict must have a 'name' item, which can be a name ID
    (int), a string, or a dictionary containing multilingual names,
    like the axis name. It translates to the ValueNameID field.

    Optionally the value dict can contain a 'flags' item. It maps to
    the AxisValue Flags field, and will be 0 when omitted.

    The format of the AxisValue is determined by the remaining contents
    of the value dictionary:

    If the value dict contains a 'value' item, an AxisValue record
    Format 1 is created. If in addition to the 'value' item it contains
    a 'linkedValue' item, an AxisValue record Format 3 is built.

    If the value dict contains a 'nominalValue' item, an AxisValue
    record Format 2 is built. Optionally it may contain 'rangeMinValue'
    and 'rangeMaxValue' items. These map to -Infinity and +Infinity
    respectively if omitted.

    You cannot specify Format 4 AxisValue tables this way, as they are
    not tied to a single axis, and specify a name for a location that
    is defined by multiple axes values. Instead, you need to supply the
    'locations' argument.

    The optional 'locations' argument specifies AxisValue Format 4
    tables. It should be a list of dicts, where each dict has a 'name'
    item, which works just like the value dicts above, an optional
    'flags' item (defaulting to 0x0), and a 'location' dict. A
    location dict key is an axis tag, and the associated value is the
    location on the specified axis. They map to the AxisIndex and Value
    fields of the AxisValueRecord.

    Example::

        locations = [
            dict(name='Regular ABCD', location=dict(wght=300, ABCD=100)),
            dict(name='Bold ABCD XYZ', location=dict(wght=600, ABCD=200)),
        ]

    The optional 'elidedFallbackName' argument can be a name ID (int),
    a string, a dictionary containing multilingual names, or a list of
    STATNameStatements. It translates to the ElidedFallbackNameID field.

    The 'ttFont' argument must be a TTFont instance that already has a
    'name' table. If a 'STAT' table already exists, it will be
    overwritten by the newly created one.
    """
    ttFont['STAT'] = ttLib.newTable('STAT')
    statTable = ttFont['STAT'].table = ot.STAT()
    statTable.ElidedFallbackNameID = _addName(ttFont, elidedFallbackName, windows=windowsNames, mac=macNames)
    axisRecords, axisValues = _buildAxisRecords(axes, ttFont, windowsNames=windowsNames, macNames=macNames)
    if not locations:
        statTable.Version = 65537
    else:
        statTable.Version = 65538
        multiAxisValues = _buildAxisValuesFormat4(locations, axes, ttFont, windowsNames=windowsNames, macNames=macNames)
        axisValues = multiAxisValues + axisValues
    ttFont['name'].names.sort()
    axisRecordArray = ot.AxisRecordArray()
    axisRecordArray.Axis = axisRecords
    statTable.DesignAxisRecordSize = 8
    statTable.DesignAxisRecord = axisRecordArray
    statTable.DesignAxisCount = len(axisRecords)
    statTable.AxisValueCount = 0
    statTable.AxisValueArray = None
    if axisValues:
        axisValueArray = ot.AxisValueArray()
        axisValueArray.AxisValue = axisValues
        statTable.AxisValueArray = axisValueArray
        statTable.AxisValueCount = len(axisValues)