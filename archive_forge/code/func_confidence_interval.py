import numpy as np
from scipy.special import ndtri
from scipy.optimize import brentq
from ._discrete_distns import nchypergeom_fisher
from ._common import ConfidenceInterval
def confidence_interval(self, confidence_level=0.95, alternative='two-sided'):
    """
        Confidence interval for the odds ratio.

        Parameters
        ----------
        confidence_level: float
            Desired confidence level for the confidence interval.
            The value must be given as a fraction between 0 and 1.
            Default is 0.95 (meaning 95%).

        alternative : {'two-sided', 'less', 'greater'}, optional
            The alternative hypothesis of the hypothesis test to which the
            confidence interval corresponds. That is, suppose the null
            hypothesis is that the true odds ratio equals ``OR`` and the
            confidence interval is ``(low, high)``. Then the following options
            for `alternative` are available (default is 'two-sided'):

            * 'two-sided': the true odds ratio is not equal to ``OR``. There
              is evidence against the null hypothesis at the chosen
              `confidence_level` if ``high < OR`` or ``low > OR``.
            * 'less': the true odds ratio is less than ``OR``. The ``low`` end
              of the confidence interval is 0, and there is evidence against
              the null hypothesis at  the chosen `confidence_level` if
              ``high < OR``.
            * 'greater': the true odds ratio is greater than ``OR``.  The
              ``high`` end of the confidence interval is ``np.inf``, and there
              is evidence against the null hypothesis at the chosen
              `confidence_level` if ``low > OR``.

        Returns
        -------
        ci : ``ConfidenceInterval`` instance
            The confidence interval, represented as an object with
            attributes ``low`` and ``high``.

        Notes
        -----
        When `kind` is ``'conditional'``, the limits of the confidence
        interval are the conditional "exact confidence limits" as described
        by Fisher [1]_. The conditional odds ratio and confidence interval are
        also discussed in Section 4.1.2 of the text by Sahai and Khurshid [2]_.

        When `kind` is ``'sample'``, the confidence interval is computed
        under the assumption that the logarithm of the odds ratio is normally
        distributed with standard error given by::

            se = sqrt(1/a + 1/b + 1/c + 1/d)

        where ``a``, ``b``, ``c`` and ``d`` are the elements of the
        contingency table.  (See, for example, [2]_, section 3.1.3.2,
        or [3]_, section 2.3.3).

        References
        ----------
        .. [1] R. A. Fisher (1935), The logic of inductive inference,
               Journal of the Royal Statistical Society, Vol. 98, No. 1,
               pp. 39-82.
        .. [2] H. Sahai and A. Khurshid (1996), Statistics in Epidemiology:
               Methods, Techniques, and Applications, CRC Press LLC, Boca
               Raton, Florida.
        .. [3] Alan Agresti, An Introduction to Categorical Data Analysis
               (second edition), Wiley, Hoboken, NJ, USA (2007).
        """
    if alternative not in ['two-sided', 'less', 'greater']:
        raise ValueError("`alternative` must be 'two-sided', 'less' or 'greater'.")
    if confidence_level < 0 or confidence_level > 1:
        raise ValueError('confidence_level must be between 0 and 1')
    if self._kind == 'conditional':
        ci = self._conditional_odds_ratio_ci(confidence_level, alternative)
    else:
        ci = self._sample_odds_ratio_ci(confidence_level, alternative)
    return ci