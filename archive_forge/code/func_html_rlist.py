import jinja2  # type: ignore
from rpy2.robjects import (vectors,
from rpy2 import rinterface
from rpy2.robjects.packages import SourceCode
from rpy2.robjects.packages import wherefrom
from IPython import get_ipython  # type: ignore
def html_rlist(vector, display_nrowmax=10, size_tail=2, table_class='rpy2_table'):
    rg = range(max(0, len(vector) - size_tail), len(vector))
    html = template_vector_vertical.render({'table_class': table_class, 'clsname': type(vector).__name__, 'vector': vector, 'has_vector_names': vector.names is not rinterface.NULL, 'display_nrowmax': min(display_nrowmax, len(vector)), 'size_tail': size_tail, 'elt_i_tail': rg})
    return html