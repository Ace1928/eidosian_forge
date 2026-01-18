from tensorflow.python.util import tf_export
def get_doc_sources(api_name):
    """Get a map from module to a DocSource object.

  Args:
    api_name: API you want to generate (e.g. `tensorflow` or `estimator`).

  Returns:
    Map from module name to DocSource object.
  """
    if api_name == tf_export.TENSORFLOW_API_NAME:
        return _TENSORFLOW_DOC_SOURCES
    if api_name == tf_export.ESTIMATOR_API_NAME:
        return _ESTIMATOR_DOC_SOURCES
    if api_name == tf_export.KERAS_API_NAME:
        return _KERAS_DOC_SOURCES
    return {}