import jinja2  # type: ignore
from rpy2.robjects import (vectors,
from rpy2 import rinterface
from rpy2.robjects.packages import SourceCode
from rpy2.robjects.packages import wherefrom
from IPython import get_ipython  # type: ignore
def html_ridentifiedobject(obj):
    d = _dict_ridentifiedobject(obj)
    html = template_ridentifiedobject.render(d)
    return html