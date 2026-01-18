from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import (
def check_news(news, revisions, updates, impact_dates, impacted_variables, revisions_index, updates_index, revision_impacts, update_impacts, prev_impacted_forecasts, post_impacted_forecasts, update_forecasts, update_realized, news_desired, weights):
    news.tolerance = -1e-10
    check_impact_indices(news, impact_dates, impacted_variables)
    check_revision_indices(news, revisions_index)
    check_update_indices(news, updates_index)
    check_news_indices(news, updates_index, impact_dates)
    if updates:
        assert_allclose(news.update_impacts, update_impacts, atol=1e-12)
    else:
        assert_(np.all(news.update_impacts.isnull()))
    if revisions:
        assert_allclose(news.revision_impacts, revision_impacts, atol=1e-12)
    else:
        assert_(news.news_results.revision_impacts is None)
        assert_(np.all(news.revision_impacts.isnull()))
    total_impacts = news.revision_impacts.astype(float).fillna(0) + news.update_impacts.astype(float).fillna(0)
    assert_allclose(news.total_impacts, total_impacts, atol=1e-12)
    assert_allclose(news.prev_impacted_forecasts, prev_impacted_forecasts, atol=1e-12)
    assert_allclose(news.post_impacted_forecasts, post_impacted_forecasts, atol=1e-12)
    assert_allclose(news.update_forecasts, update_forecasts, atol=1e-12)
    assert_allclose(news.update_realized, update_realized, atol=1e-12)
    assert_allclose(news.news, news_desired, atol=1e-12)
    assert_allclose(news.weights, weights, atol=1e-12)
    assert_equal(news.data_revisions.columns.tolist(), ['revised', 'observed (prev)', 'detailed impacts computed'])
    assert_equal(news.data_revisions.index.names, ['revision date', 'revised variable'])
    assert_(news.data_revisions.index.equals(revisions_index))
    assert_equal(news.data_updates.columns.tolist(), ['observed', 'forecast (prev)'])
    assert_equal(news.data_updates.index.names, ['update date', 'updated variable'])
    assert_(news.data_updates.index.equals(news.news.index))
    assert_allclose(news.data_updates['forecast (prev)'], news.update_forecasts, atol=1e-12)
    assert_allclose(news.data_updates['observed'], news.update_realized, atol=1e-12)
    details_by_impact = news.details_by_impact
    desired = ['observed', 'forecast (prev)', 'news', 'weight', 'impact']
    assert_equal(details_by_impact.columns.tolist(), desired)
    desired = ['impact date', 'impacted variable', 'update date', 'updated variable']
    assert_equal(details_by_impact.index.names, desired)
    if updates:
        actual = news.details_by_impact['forecast (prev)'].drop_duplicates().reset_index([0, 1])['forecast (prev)']
        assert_allclose(actual, news.update_forecasts, atol=1e-12)
        actual = news.details_by_impact['observed'].drop_duplicates().reset_index([0, 1])['observed']
        assert_allclose(actual, news.update_realized, atol=1e-12)
        actual = news.details_by_impact['news'].drop_duplicates().reset_index([0, 1])['news']
        assert_allclose(actual, news.news, atol=1e-12)
        assert_allclose(details_by_impact['weight'].unstack([0, 1]), news.weights, atol=1e-12)
        actual = news.details_by_impact['impact'].unstack([2, 3]).sum(axis=1).unstack(1)
        assert_allclose(actual, news.update_impacts, atol=1e-12)
    details_by_update = news.details_by_update
    desired = ['news', 'weight', 'impact']
    assert_equal(details_by_update.columns.tolist(), desired)
    desired = ['update date', 'updated variable', 'observed', 'forecast (prev)', 'impact date', 'impacted variable']
    assert_equal(details_by_update.index.names, desired)
    if updates:
        actual = news.details_by_update['news'].drop_duplicates().reset_index([2, 3, 4, 5])['news']
        assert_allclose(actual, news.news, atol=1e-12)
        assert_allclose(news.details_by_update['weight'].unstack([4, 5]), news.weights, atol=1e-12)
        actual = news.details_by_update['impact'].unstack([4, 5]).sum(axis=0).unstack(1)
        assert_allclose(actual, news.update_impacts, atol=1e-12)
    impacts = news.impacts
    desired = ['estimate (prev)', 'impact of revisions', 'impact of news', 'total impact', 'estimate (new)']
    assert_equal(impacts.columns.tolist(), desired)
    desired = ['impact date', 'impacted variable']
    assert_equal(impacts.index.names, desired)
    assert_allclose(impacts.loc[:, 'estimate (prev)'], news.prev_impacted_forecasts.stack(**FUTURE_STACK), atol=1e-12)
    assert_allclose(impacts.loc[:, 'impact of revisions'], news.revision_impacts.astype(float).fillna(0).stack(**FUTURE_STACK), atol=1e-12)
    assert_allclose(impacts.loc[:, 'impact of news'], news.update_impacts.astype(float).fillna(0).stack(**FUTURE_STACK), atol=1e-12)
    assert_allclose(impacts.loc[:, 'total impact'], news.total_impacts.stack(**FUTURE_STACK), atol=1e-12)
    assert_allclose(impacts.loc[:, 'estimate (new)'], news.post_impacted_forecasts.stack(**FUTURE_STACK), atol=1e-12)