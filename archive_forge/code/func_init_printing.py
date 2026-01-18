import jinja2  # type: ignore
from rpy2.robjects import (vectors,
from rpy2 import rinterface
from rpy2.robjects.packages import SourceCode
from rpy2.robjects.packages import wherefrom
from IPython import get_ipython  # type: ignore
def init_printing():
    ip = get_ipython()
    html_f = ip.display_formatter.formatters['text/html']
    html_f.for_type(vectors.Vector, html_vector_horizontal)
    html_f.for_type(vectors.ListVector, html_rlist)
    html_f.for_type(vectors.DataFrame, html_rdataframe)
    html_f.for_type(RObject, html_ridentifiedobject)
    html_f.for_type(RS4, html_rs4)
    html_f.for_type(SignatureTranslatedFunction, html_ridentifiedobject)
    html_f.for_type(SourceCode, html_sourcecode)