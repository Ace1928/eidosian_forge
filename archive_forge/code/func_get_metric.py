import tree
from keras.src import backend
from keras.src import losses as losses_module
from keras.src import metrics as metrics_module
from keras.src import ops
from keras.src.utils.naming import get_object_name
def get_metric(identifier, y_true, y_pred):
    if identifier is None:
        return None
    if str(identifier).lower() not in ['accuracy', 'acc']:
        metric_obj = metrics_module.get(identifier)
    else:
        is_binary, is_sparse_categorical = is_binary_or_sparse_categorical(y_true, y_pred)
        if is_binary:
            metric_obj = metrics_module.BinaryAccuracy(name=str(identifier))
        elif is_sparse_categorical:
            metric_obj = metrics_module.SparseCategoricalAccuracy(name=str(identifier))
        else:
            metric_obj = metrics_module.CategoricalAccuracy(name=str(identifier))
    if isinstance(identifier, str):
        metric_name = identifier
    else:
        metric_name = get_object_name(metric_obj)
    if not isinstance(metric_obj, metrics_module.Metric):
        metric_obj = metrics_module.MeanMetricWrapper(metric_obj)
    metric_obj.name = metric_name
    return metric_obj