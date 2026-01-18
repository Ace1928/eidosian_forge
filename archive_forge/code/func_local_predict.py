from . import custom_code_utils
from . import prediction_utils
def local_predict(model_dir=None, signature_name=None, instances=None, framework=None, **kwargs):
    """Run a prediction locally."""
    framework = framework or prediction_utils.TENSORFLOW_FRAMEWORK_NAME
    client = create_client(framework, model_dir, **kwargs)
    model = create_model(client, model_dir, framework)
    if prediction_utils.should_base64_decode(framework, model, signature_name):
        instances = prediction_utils.decode_base64(instances)
    predictions = model.predict(instances, signature_name=signature_name)
    return {'predictions': list(predictions)}