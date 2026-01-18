import argparse
import builtins
import functools
import importlib
import json
import os
import sys
import mlflow
from mlflow.models.model import MLMODEL_FILE_NAME, Model
from mlflow.pyfunc import MAIN
from mlflow.utils._spark_utils import _prepare_subprocess_environ_for_creating_local_spark_session
from mlflow.utils.exception_utils import get_stacktrace
from mlflow.utils.file_utils import write_to
from mlflow.utils.requirements_utils import (
def store_imported_modules(cap_cm, model_path, flavor, output_file, error_file=None):
    if os.path.isdir(model_path) and MLMODEL_FILE_NAME in os.listdir(model_path):
        mlflow_model = Model.load(model_path)
        pyfunc_conf = mlflow_model.flavors.get(mlflow.pyfunc.FLAVOR_NAME)
        input_example = mlflow_model.load_input_example(model_path)
        params = mlflow_model.load_input_example_params(model_path)
        loader_module = importlib.import_module(pyfunc_conf[MAIN])
        original = loader_module._load_pyfunc

        @functools.wraps(original)
        def _load_pyfunc_patch(*args, **kwargs):
            with cap_cm:
                model = original(*args, **kwargs)
                if input_example is not None:
                    try:
                        model.predict(input_example, params=params)
                    except Exception as e:
                        if error_file:
                            stack_trace = get_stacktrace(e)
                            write_to(error_file, 'Failed to run predict on input_example, dependencies introduced in predict are not captured.\n' + stack_trace)
                        else:
                            raise e
                return model
        loader_module._load_pyfunc = _load_pyfunc_patch
        try:
            mlflow.pyfunc.load_model(model_path)
        finally:
            loader_module._load_pyfunc = original
    else:
        with cap_cm:
            importlib.import_module(f'mlflow.{flavor}')._load_pyfunc(model_path)
    write_to(output_file, '\n'.join(cap_cm.imported_modules))