import numpy as np
from scipy import stats
from scipy.special import ncfdtrinc
from statsmodels.stats.power import ncf_cdf, ncf_ppf
from statsmodels.stats.robust_compare import TrimmedMean, scale_transform
from statsmodels.tools.testing import Holder
from statsmodels.stats.base import HolderTuple
def equivalence_scale_oneway(data, equiv_margin, method='bf', center='median', transform='abs', trim_frac_mean=0.0, trim_frac_anova=0.0):
    """Oneway Anova test for equivalence of scale, variance or dispersion

    This hypothesis test performs a oneway equivalence anova test on
    transformed data.

    Note, the interpretation of the equivalence margin `equiv_margin` will
    depend on the transformation of the data. Transformations like
    absolute deviation are not scaled to correspond to the variance under
    normal distribution.

    Parameters
    ----------
    data : tuple of array_like or DataFrame or Series
        Data for k independent samples, with k >= 2. The data can be provided
        as a tuple or list of arrays or in long format with outcome
        observations in ``data`` and group membership in ``groups``.
    equiv_margin : float
        Equivalence margin in terms of effect size. Effect size can be chosen
        with `margin_type`. default is squared Cohen's f.
    method : {"unequal", "equal" or "bf"}
        How to treat heteroscedasticity across samples. This is used as
        `use_var` option in `anova_oneway` and refers to the variance of the
        transformed data, i.e. assumption is on 4th moment if squares are used
        as transform.
        Three approaches are available:

        "unequal" : Variances are not assumed to be equal across samples.
            Heteroscedasticity is taken into account with Welch Anova and
            Satterthwaite-Welch degrees of freedom.
            This is the default.
        "equal" : Variances are assumed to be equal across samples.
            This is the standard Anova.
        "bf" : Variances are not assumed to be equal across samples.
            The method is Browne-Forsythe (1971) for testing equality of means
            with the corrected degrees of freedom by Merothra. The original BF
            degrees of freedom are available as additional attributes in the
            results instance, ``df_denom2`` and ``p_value2``.
    center : "median", "mean", "trimmed" or float
        Statistic used for centering observations. If a float, then this
        value is used to center. Default is median.
    transform : "abs", "square" or callable
        Transformation for the centered observations. If a callable, then this
        function is called on the centered data.
        Default is absolute value.
    trim_frac_mean : float in [0, 0.5)
        Trim fraction for the trimmed mean when `center` is "trimmed"
    trim_frac_anova : float in [0, 0.5)
        Optional trimming for Anova with trimmed mean and Winsorized variances.
        With the default trim_frac equal to zero, the oneway Anova statistics
        are computed without trimming. If `trim_frac` is larger than zero,
        then the largest and smallest observations in each sample are trimmed.
        see ``trim_frac`` option in `anova_oneway`

    Returns
    -------
    results : instance of HolderTuple class
        The two main attributes are test statistic `statistic` and p-value
        `pvalue`.

    See Also
    --------
    anova_oneway
    scale_transform
    equivalence_oneway
    """
    data = map(np.asarray, data)
    xxd = [scale_transform(x, center=center, transform=transform, trim_frac=trim_frac_mean) for x in data]
    res = equivalence_oneway(xxd, equiv_margin, use_var=method, welch_correction=True, trim_frac=trim_frac_anova)
    res.x_transformed = xxd
    return res