from fontTools.ttLib.ttVisitor import TTVisitor
import fontTools.ttLib as ttLib
import fontTools.ttLib.tables.otBase as otBase
import fontTools.ttLib.tables.otTables as otTables
from fontTools.cffLib import VarStoreData
import fontTools.cffLib.specializer as cffSpecializer
from fontTools.varLib import builder  # for VarData.calculateNumShorts
from fontTools.misc.fixedTools import otRound
from fontTools.ttLib.tables._g_l_y_f import VarComponentFlags
def scale_upem(font, new_upem):
    """Change the units-per-EM of font to the new value."""
    upem = font['head'].unitsPerEm
    visitor = ScalerVisitor(new_upem / upem)
    visitor.visit(font)