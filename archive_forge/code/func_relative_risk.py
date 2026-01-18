import operator
from dataclasses import dataclass
import numpy as np
from scipy.special import ndtri
from ._common import ConfidenceInterval
def relative_risk(exposed_cases, exposed_total, control_cases, control_total):
    """
    Compute the relative risk (also known as the risk ratio).

    This function computes the relative risk associated with a 2x2
    contingency table ([1]_, section 2.2.3; [2]_, section 3.1.2). Instead
    of accepting a table as an argument, the individual numbers that are
    used to compute the relative risk are given as separate parameters.
    This is to avoid the ambiguity of which row or column of the contingency
    table corresponds to the "exposed" cases and which corresponds to the
    "control" cases.  Unlike, say, the odds ratio, the relative risk is not
    invariant under an interchange of the rows or columns.

    Parameters
    ----------
    exposed_cases : nonnegative int
        The number of "cases" (i.e. occurrence of disease or other event
        of interest) among the sample of "exposed" individuals.
    exposed_total : positive int
        The total number of "exposed" individuals in the sample.
    control_cases : nonnegative int
        The number of "cases" among the sample of "control" or non-exposed
        individuals.
    control_total : positive int
        The total number of "control" individuals in the sample.

    Returns
    -------
    result : instance of `~scipy.stats._result_classes.RelativeRiskResult`
        The object has the float attribute ``relative_risk``, which is::

            rr = (exposed_cases/exposed_total) / (control_cases/control_total)

        The object also has the method ``confidence_interval`` to compute
        the confidence interval of the relative risk for a given confidence
        level.

    See Also
    --------
    odds_ratio

    Notes
    -----
    The R package epitools has the function `riskratio`, which accepts
    a table with the following layout::

                        disease=0   disease=1
        exposed=0 (ref)    n00         n01
        exposed=1          n10         n11

    With a 2x2 table in the above format, the estimate of the CI is
    computed by `riskratio` when the argument method="wald" is given,
    or with the function `riskratio.wald`.

    For example, in a test of the incidence of lung cancer among a
    sample of smokers and nonsmokers, the "exposed" category would
    correspond to "is a smoker" and the "disease" category would
    correspond to "has or had lung cancer".

    To pass the same data to ``relative_risk``, use::

        relative_risk(n11, n10 + n11, n01, n00 + n01)

    .. versionadded:: 1.7.0

    References
    ----------
    .. [1] Alan Agresti, An Introduction to Categorical Data Analysis
           (second edition), Wiley, Hoboken, NJ, USA (2007).
    .. [2] Hardeo Sahai and Anwer Khurshid, Statistics in Epidemiology,
           CRC Press LLC, Boca Raton, FL, USA (1996).

    Examples
    --------
    >>> from scipy.stats.contingency import relative_risk

    This example is from Example 3.1 of [2]_.  The results of a heart
    disease study are summarized in the following table::

                 High CAT   Low CAT    Total
                 --------   -------    -----
        CHD         27         44        71
        No CHD      95        443       538

        Total      122        487       609

    CHD is coronary heart disease, and CAT refers to the level of
    circulating catecholamine.  CAT is the "exposure" variable, and
    high CAT is the "exposed" category. So the data from the table
    to be passed to ``relative_risk`` is::

        exposed_cases = 27
        exposed_total = 122
        control_cases = 44
        control_total = 487

    >>> result = relative_risk(27, 122, 44, 487)
    >>> result.relative_risk
    2.4495156482861398

    Find the confidence interval for the relative risk.

    >>> result.confidence_interval(confidence_level=0.95)
    ConfidenceInterval(low=1.5836990926700116, high=3.7886786315466354)

    The interval does not contain 1, so the data supports the statement
    that high CAT is associated with greater risk of CHD.
    """
    exposed_cases = _validate_int(exposed_cases, 0, 'exposed_cases')
    exposed_total = _validate_int(exposed_total, 1, 'exposed_total')
    control_cases = _validate_int(control_cases, 0, 'control_cases')
    control_total = _validate_int(control_total, 1, 'control_total')
    if exposed_cases > exposed_total:
        raise ValueError('exposed_cases must not exceed exposed_total.')
    if control_cases > control_total:
        raise ValueError('control_cases must not exceed control_total.')
    if exposed_cases == 0 and control_cases == 0:
        rr = np.nan
    elif exposed_cases == 0:
        rr = 0.0
    elif control_cases == 0:
        rr = np.inf
    else:
        p1 = exposed_cases / exposed_total
        p2 = control_cases / control_total
        rr = p1 / p2
    return RelativeRiskResult(relative_risk=rr, exposed_cases=exposed_cases, exposed_total=exposed_total, control_cases=control_cases, control_total=control_total)