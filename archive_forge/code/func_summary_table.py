import warnings
from statsmodels.compat.pandas import Appender
from statsmodels.compat.python import lzip
from collections import defaultdict
import numpy as np
from statsmodels.graphics._regressionplots_doc import _plot_influence_doc
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import maybe_unwrap_results
def summary_table(self, float_fmt='%6.3f'):
    """create a summary table with all influence and outlier measures

        This does currently not distinguish between statistics that can be
        calculated from the original regression results and for which a
        leave-one-observation-out loop is needed

        Returns
        -------
        res : SimpleTable
           SimpleTable instance with the results, can be printed

        Notes
        -----
        This also attaches table_data to the instance.
        """
    table_raw = [('obs', np.arange(self.nobs)), ('endog', self.endog), ('fitted\nvalue', self.results.fittedvalues), ("Cook's\nd", self.cooks_distance[0]), ('student.\nresidual', self.resid_studentized_internal), ('hat diag', self.hat_matrix_diag), ('dffits \ninternal', self.dffits_internal[0]), ('ext.stud.\nresidual', self.resid_studentized_external), ('dffits', self.dffits[0])]
    colnames, data = lzip(*table_raw)
    data = np.column_stack(data)
    self.table_data = data
    from copy import deepcopy
    from statsmodels.iolib.table import SimpleTable, default_html_fmt
    from statsmodels.iolib.tableformatting import fmt_base
    fmt = deepcopy(fmt_base)
    fmt_html = deepcopy(default_html_fmt)
    fmt['data_fmts'] = ['%4d'] + [float_fmt] * (data.shape[1] - 1)
    return SimpleTable(data, headers=colnames, txt_fmt=fmt, html_fmt=fmt_html)