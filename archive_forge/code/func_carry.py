from . import _catboost
def carry(model, features):
    """
    Parameters
    ----------
    model :
        CatBoost / CatBoostClassifier / CatBoostRanker / CatBoostRegressor model
    features :
        must be a dict mapping strings (factor names) or integers (flat indexes) into floats
        NOTE: values in a dict can be lists of floats, but in this case they must all be the same size
    """
    model = model.copy()
    assert type(features) is dict
    factor_ids = []
    factor_values = []
    for factor_id, factor_value in features.items():
        assert type(factor_id) is str or type(factor_id) is int
        assert type(factor_value) is float or type(factor_value) is int or type(factor_value) is list
        if type(factor_value) is list:
            for value in factor_value:
                assert type(value) is float or type(value) is int
        else:
            factor_value = [factor_value]
        if len(factor_values):
            assert len(factor_values[0]) == len(factor_value)
            assert type(factor_ids[0]) == type(factor_id)
        factor_ids.append(factor_id)
        factor_values.append(factor_value)
    if len(factor_ids):
        if type(factor_ids[0]) is int:
            _carry_by_index(model._object, factor_ids, factor_values)
        else:
            _carry_by_name(model._object, factor_ids, factor_values)
    return model