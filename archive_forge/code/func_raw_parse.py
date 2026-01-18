from pythran.openmp import GatherOMPData
from pythran.syntax import check_syntax
from pythran.transformations import ExtractDocStrings, HandleImport
import gast as ast
import re
def raw_parse(code):
    code = re.sub('(\\s*)#\\s*(omp\\s[^\\n]+)', '\\1"\\2"', code)
    return ast.parse(code)