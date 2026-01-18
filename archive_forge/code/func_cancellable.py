from tensorflow.python import pywrap_tfe
def cancellable(*args, **kwargs):
    with CancellationManagerContext(self):
        return concrete_function(*args, **kwargs)