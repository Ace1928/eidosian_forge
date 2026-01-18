import warnings
import numpy as np
from numpy.linalg import eigh, inv, norm, matrix_rank
import pandas as pd
from scipy.optimize import minimize
from statsmodels.tools.decorators import cache_readonly
from statsmodels.base.model import Model
from statsmodels.iolib import summary2
from statsmodels.graphics.utils import _import_mpl
from .factor_rotation import rotate_factors, promax
def get_loadings_frame(self, style='display', sort_=True, threshold=0.3, highlight_max=True, color_max='yellow', decimals=None):
    """get loadings matrix as DataFrame or pandas Styler

        Parameters
        ----------
        style : 'display' (default), 'raw' or 'strings'
            Style to use for display

            * 'raw' returns just a DataFrame of the loadings matrix, no options are
               applied
            * 'display' add sorting and styling as defined by other keywords
            * 'strings' returns a DataFrame with string elements with optional sorting
               and suppressing small loading coefficients.

        sort_ : bool
            If True, then the rows of the DataFrame is sorted by contribution of each
            factor. applies if style is either 'display' or 'strings'
        threshold : float
            If the threshold is larger than zero, then loading coefficients are
            either colored white (if style is 'display') or replace by empty
            string (if style is 'strings').
        highlight_max : bool
            This add a background color to the largest coefficient in each row.
        color_max : html color
            default is 'yellow'. color for background of row maximum
        decimals : None or int
            If None, then pandas default precision applies. Otherwise values are
            rounded to the specified decimals. If style is 'display', then the
            underlying dataframe is not changed. If style is 'strings', then
            values are rounded before conversion to strings.

        Returns
        -------
        loadings : DataFrame or pandas Styler instance
            The return is a pandas Styler instance, if style is 'display' and
            at least one of highlight_max, threshold or decimals is applied.
            Otherwise, the returned loadings is a DataFrame.

        Examples
        --------
        >>> mod = Factor(df, 3, smc=True)
        >>> res = mod.fit()
        >>> res.get_loadings_frame(style='display', decimals=3, threshold=0.2)

        To get a sorted DataFrame, all styling options need to be turned off:

        >>> df_sorted = res.get_loadings_frame(style='display',
        ...             highlight_max=False, decimals=None, threshold=0)

        Options except for highlighting are available for plain test or Latex
        usage:

        >>> lds = res_u.get_loadings_frame(style='strings', decimals=3,
        ...                                threshold=0.3)
        >>> print(lds.to_latex())
        """
    loadings_df = pd.DataFrame(self.loadings, columns=['factor %d' % i for i in range(self.loadings.shape[1])], index=self.endog_names)
    if style not in ['raw', 'display', 'strings']:
        msg = "style has to be one of 'raw', 'display', 'strings'"
        raise ValueError(msg)
    if style == 'raw':
        return loadings_df
    if sort_ is True:
        loadings_df2 = loadings_df.copy()
        n_f = len(loadings_df2)
        high = np.abs(loadings_df2.values).argmax(1)
        loadings_df2['high'] = high
        loadings_df2['largest'] = np.abs(loadings_df.values[np.arange(n_f), high])
        loadings_df2.sort_values(by=['high', 'largest'], ascending=[True, False], inplace=True)
        loadings_df = loadings_df2.drop(['high', 'largest'], axis=1)
    if style == 'display':
        sty = None
        if threshold > 0:

            def color_white_small(val):
                """
                    Takes a scalar and returns a string with
                    the css property `'color: white'` for small values, black otherwise.

                    takes threshold from outer scope
                    """
                color = 'white' if np.abs(val) < threshold else 'black'
                return 'color: %s' % color
            try:
                sty = loadings_df.style.map(color_white_small)
            except AttributeError:
                sty = loadings_df.style.applymap(color_white_small)
        if highlight_max is True:

            def highlight_max(s):
                """
                    highlight the maximum in a Series yellow.
                    """
                s = np.abs(s)
                is_max = s == s.max()
                return ['background-color: ' + color_max if v else '' for v in is_max]
            if sty is None:
                sty = loadings_df.style
            sty = sty.apply(highlight_max, axis=1)
        if decimals is not None:
            if sty is None:
                sty = loadings_df.style
            sty.format('{:.%sf}' % decimals)
        if sty is None:
            return loadings_df
        else:
            return sty
    if style == 'strings':
        ld = loadings_df
        if decimals is not None:
            ld = ld.round(decimals)
        ld = ld.astype(str)
        if threshold > 0:
            ld[loadings_df.abs() < threshold] = ''
        return ld