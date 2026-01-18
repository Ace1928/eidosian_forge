from numpy.testing import assert_equal
import numpy as np
def summary_coeff(self):
    from statsmodels.iolib import SimpleTable
    params_arr = self.params.reshape(self.nlevel1, self.nlevel2)
    stubs = self.d1_labels
    headers = self.d2_labels
    title = 'Estimated Coefficients by factors'
    table_fmt = dict(data_fmts=['%#10.4g'] * self.nlevel2)
    return SimpleTable(params_arr, headers, stubs, title=title, txt_fmt=table_fmt)