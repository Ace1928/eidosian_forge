from fontTools.misc.dictTools import hashdict
from fontTools.misc.intTools import bit_count
from fontTools.ttLib import newTable
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.ttVisitor import TTVisitor
from fontTools.otlLib.builder import buildLookup, buildSingleSubstSubtable
from collections import OrderedDict
from .errors import VarLibError, VarLibValidationError
def buildGSUB():
    """Build a GSUB table from scratch."""
    fontTable = newTable('GSUB')
    gsub = fontTable.table = ot.GSUB()
    gsub.Version = 65537
    gsub.ScriptList = ot.ScriptList()
    gsub.ScriptList.ScriptRecord = []
    gsub.FeatureList = ot.FeatureList()
    gsub.FeatureList.FeatureRecord = []
    gsub.LookupList = ot.LookupList()
    gsub.LookupList.Lookup = []
    srec = ot.ScriptRecord()
    srec.ScriptTag = 'DFLT'
    srec.Script = ot.Script()
    srec.Script.DefaultLangSys = None
    srec.Script.LangSysRecord = []
    srec.Script.LangSysCount = 0
    langrec = ot.LangSysRecord()
    langrec.LangSys = ot.LangSys()
    langrec.LangSys.ReqFeatureIndex = 65535
    langrec.LangSys.FeatureIndex = []
    srec.Script.DefaultLangSys = langrec.LangSys
    gsub.ScriptList.ScriptRecord.append(srec)
    gsub.ScriptList.ScriptCount = 1
    gsub.FeatureVariations = None
    return fontTable