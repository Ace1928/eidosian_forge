import jinja2  # type: ignore
from rpy2.robjects import (vectors,
from rpy2 import rinterface
from rpy2.robjects.packages import SourceCode
from rpy2.robjects.packages import wherefrom
from IPython import get_ipython  # type: ignore
def html_rdataframe(dataf, display_nrowmax=10, display_ncolmax=6, size_coltail=2, size_rowtail=2, table_class='rpy2_table'):
    html = template_dataframe.render({'dataf': StrDataFrame(dataf), 'table_class': table_class, 'has_rownames': dataf.rownames is not None, 'clsname': type(dataf).__name__, 'display_nrowmax': min(display_nrowmax, dataf.nrow), 'display_ncolmax': min(display_ncolmax, dataf.ncol), 'col_i_tail': range(max(0, dataf.ncol - size_coltail), dataf.ncol), 'row_i_tail': range(max(0, dataf.nrow - size_rowtail), dataf.nrow), 'size_coltail': size_coltail, 'size_rowtail': size_rowtail})
    return html