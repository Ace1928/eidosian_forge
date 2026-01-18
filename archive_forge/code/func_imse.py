import numpy as np
from . import kernels
from ._kernel_base import GenericKDE, EstimatorSettings, gpke, \
def imse(self, bw):
    """
        The integrated mean square error for the conditional KDE.

        Parameters
        ----------
        bw : array_like
            The bandwidth parameter(s).

        Returns
        -------
        CV : float
            The cross-validation objective function.

        Notes
        -----
        For more details see pp. 156-166 in [1]_. For details on how to
        handle the mixed variable types see [2]_.

        The formula for the cross-validation objective function for mixed
        variable types is:

        .. math:: CV(h,\\lambda)=\\frac{1}{n}\\sum_{l=1}^{n}
            \\frac{G_{-l}(X_{l})}{\\left[\\mu_{-l}(X_{l})\\right]^{2}}-
            \\frac{2}{n}\\sum_{l=1}^{n}\\frac{f_{-l}(X_{l},Y_{l})}{\\mu_{-l}(X_{l})}

        where

        .. math:: G_{-l}(X_{l}) = n^{-2}\\sum_{i\\neq l}\\sum_{j\\neq l}
                        K_{X_{i},X_{l}} K_{X_{j},X_{l}}K_{Y_{i},Y_{j}}^{(2)}

        where :math:`K_{X_{i},X_{l}}` is the multivariate product kernel and
        :math:`\\mu_{-l}(X_{l})` is the leave-one-out estimator of the pdf.

        :math:`K_{Y_{i},Y_{j}}^{(2)}` is the convolution kernel.

        The value of the function is minimized by the ``_cv_ls`` method of the
        `GenericKDE` class to return the bw estimates that minimize the
        distance between the estimated and "true" probability density.

        References
        ----------
        .. [1] Racine, J., Li, Q. Nonparametric econometrics: theory and
                practice. Princeton University Press. (2007)
        .. [2] Racine, J., Li, Q. "Nonparametric Estimation of Distributions
                with Categorical and Continuous Data." Working Paper. (2000)
        """
    zLOO = LeaveOneOut(self.data)
    CV = 0
    nobs = float(self.nobs)
    expander = np.ones((self.nobs - 1, 1))
    for ii, Z in enumerate(zLOO):
        X = Z[:, self.k_dep:]
        Y = Z[:, :self.k_dep]
        Ye_L = np.kron(Y, expander)
        Ye_R = np.kron(expander, Y)
        Xe_L = np.kron(X, expander)
        Xe_R = np.kron(expander, X)
        K_Xi_Xl = gpke(bw[self.k_dep:], data=Xe_L, data_predict=self.exog[ii, :], var_type=self.indep_type, tosum=False)
        K_Xj_Xl = gpke(bw[self.k_dep:], data=Xe_R, data_predict=self.exog[ii, :], var_type=self.indep_type, tosum=False)
        K2_Yi_Yj = gpke(bw[0:self.k_dep], data=Ye_L, data_predict=Ye_R, var_type=self.dep_type, ckertype='gauss_convolution', okertype='wangryzin_convolution', ukertype='aitchisonaitken_convolution', tosum=False)
        G = (K_Xi_Xl * K_Xj_Xl * K2_Yi_Yj).sum() / nobs ** 2
        f_X_Y = gpke(bw, data=-Z, data_predict=-self.data[ii, :], var_type=self.dep_type + self.indep_type) / nobs
        m_x = gpke(bw[self.k_dep:], data=-X, data_predict=-self.exog[ii, :], var_type=self.indep_type) / nobs
        CV += G / m_x ** 2 - 2 * (f_X_Y / m_x)
    return CV / nobs