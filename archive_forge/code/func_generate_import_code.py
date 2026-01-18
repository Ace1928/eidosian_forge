import deap
from copy import copy
def generate_import_code(pipeline, operators, impute=False, random_state=None):
    """Generate all library import calls for use in TPOT.export().

    Parameters
    ----------
    pipeline: List
        List of operators in the current optimized pipeline
    operators:
        List of operator class from operator library
    impute : bool
        Whether to impute new values in the feature set.
    random_state: integer or None
        Random seed in train_test_split function and exported pipeline.

    Returns
    -------
    pipeline_text: String
        The Python code that imports all required library used in the current
        optimized pipeline

    """

    def merge_imports(old_dict, new_dict):
        for key in new_dict.keys():
            if key in old_dict.keys():
                old_dict[key] = set(old_dict[key]) | set(new_dict[key])
            else:
                old_dict[key] = set(new_dict[key])
    operators_used = [x.name for x in pipeline if isinstance(x, deap.gp.Primitive)]
    pipeline_text = 'import numpy as np\nimport pandas as pd\n'
    import_relations = {op.__name__: op.import_hash for op in operators}
    flatten_list = lambda list_: [item for sublist in list_ for item in sublist]
    modules_used = [module.split('.')[0] for module in flatten_list([list(val.keys()) for val in import_relations.values()])]
    if 'imblearn' in modules_used:
        pipeline_module = 'imblearn'
    else:
        pipeline_module = 'sklearn'
    pipeline_imports = _starting_imports(operators, operators_used, pipeline_module)
    for op in operators_used:
        try:
            operator_import = import_relations[op]
            merge_imports(pipeline_imports, operator_import)
        except KeyError:
            pass
    for key in sorted(pipeline_imports.keys()):
        module_list = ', '.join(sorted(pipeline_imports[key]))
        pipeline_text += 'from {} import {}\n'.format(key, module_list)
    if pipeline_module == 'imblearn':
        pipeline_text += 'from imblearn.pipeline import make_pipeline\n'
    if impute:
        pipeline_text += 'from sklearn.impute import SimpleImputer\n'
    if random_state is not None and 'sklearn.pipeline' in pipeline_imports:
        pipeline_text += 'from tpot.export_utils import set_param_recursive\n'
    return pipeline_text