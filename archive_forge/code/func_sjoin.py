from typing import Optional
import warnings
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from geopandas import _compat as compat
from geopandas.array import _check_crs, _crs_mismatch_warn
def sjoin(left_df, right_df, how='inner', predicate='intersects', lsuffix='left', rsuffix='right', **kwargs):
    """Spatial join of two GeoDataFrames.

    See the User Guide page :doc:`../../user_guide/mergingdata` for details.


    Parameters
    ----------
    left_df, right_df : GeoDataFrames
    how : string, default 'inner'
        The type of join:

        * 'left': use keys from left_df; retain only left_df geometry column
        * 'right': use keys from right_df; retain only right_df geometry column
        * 'inner': use intersection of keys from both dfs; retain only
          left_df geometry column
    predicate : string, default 'intersects'
        Binary predicate. Valid values are determined by the spatial index used.
        You can check the valid values in left_df or right_df as
        ``left_df.sindex.valid_query_predicates`` or
        ``right_df.sindex.valid_query_predicates``
        Replaces deprecated ``op`` parameter.
    lsuffix : string, default 'left'
        Suffix to apply to overlapping column names (left GeoDataFrame).
    rsuffix : string, default 'right'
        Suffix to apply to overlapping column names (right GeoDataFrame).

    Examples
    --------
    >>> import geodatasets
    >>> chicago = geopandas.read_file(
    ...     geodatasets.get_path("geoda.chicago_health")
    ... )
    >>> groceries = geopandas.read_file(
    ...     geodatasets.get_path("geoda.groceries")
    ... ).to_crs(chicago.crs)

    >>> chicago.head()  # doctest: +SKIP
        ComAreaID  ...                                           geometry
    0         35  ...  POLYGON ((-87.60914 41.84469, -87.60915 41.844...
    1         36  ...  POLYGON ((-87.59215 41.81693, -87.59231 41.816...
    2         37  ...  POLYGON ((-87.62880 41.80189, -87.62879 41.801...
    3         38  ...  POLYGON ((-87.60671 41.81681, -87.60670 41.816...
    4         39  ...  POLYGON ((-87.59215 41.81693, -87.59215 41.816...
    [5 rows x 87 columns]

    >>> groceries.head()  # doctest: +SKIP
        OBJECTID     Ycoord  ...  Category                         geometry
    0        16  41.973266  ...       NaN  MULTIPOINT (-87.65661 41.97321)
    1        18  41.696367  ...       NaN  MULTIPOINT (-87.68136 41.69713)
    2        22  41.868634  ...       NaN  MULTIPOINT (-87.63918 41.86847)
    3        23  41.877590  ...       new  MULTIPOINT (-87.65495 41.87783)
    4        27  41.737696  ...       NaN  MULTIPOINT (-87.62715 41.73623)
    [5 rows x 8 columns]

    >>> groceries_w_communities = geopandas.sjoin(groceries, chicago)
    >>> groceries_w_communities.head()  # doctest: +SKIP
            OBJECTID     Ycoord     Xcoord  ... GonorrF GonorrM Tuberc
    0          16  41.973266 -87.657073  ...   170.8   468.7   13.6
    87        365  41.961707 -87.654058  ...   170.8   468.7   13.6
    90        373  41.963131 -87.656352  ...   170.8   468.7   13.6
    140       582  41.969131 -87.674882  ...   170.8   468.7   13.6
    1          18  41.696367 -87.681315  ...   800.5   741.1    2.6
    [5 rows x 95 columns]

    See also
    --------
    overlay : overlay operation resulting in a new geometry
    GeoDataFrame.sjoin : equivalent method

    Notes
    -----
    Every operation in GeoPandas is planar, i.e. the potential third
    dimension is not taken into account.
    """
    if 'op' in kwargs:
        op = kwargs.pop('op')
        deprecation_message = 'The `op` parameter is deprecated and will be removed in a future release. Please use the `predicate` parameter instead.'
        if predicate != 'intersects' and op != predicate:
            override_message = f'A non-default value for `predicate` was passed (got `predicate="{predicate}"` in combination with `op="{op}"`). The value of `predicate` will be overridden by the value of `op`, , which may result in unexpected behavior.\n{deprecation_message}'
            warnings.warn(override_message, UserWarning, stacklevel=4)
        else:
            warnings.warn(deprecation_message, FutureWarning, stacklevel=4)
        predicate = op
    if kwargs:
        first = next(iter(kwargs.keys()))
        raise TypeError(f"sjoin() got an unexpected keyword argument '{first}'")
    _basic_checks(left_df, right_df, how, lsuffix, rsuffix)
    indices = _geom_predicate_query(left_df, right_df, predicate)
    joined = _frame_join(indices, left_df, right_df, how, lsuffix, rsuffix)
    return joined