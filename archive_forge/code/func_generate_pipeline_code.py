import deap
from copy import copy
def generate_pipeline_code(pipeline_tree, operators):
    """Generate code specific to the construction of the sklearn Pipeline.

    Parameters
    ----------
    pipeline_tree: list
        List of operators in the current optimized pipeline

    Returns
    -------
    Source code for the sklearn pipeline

    """
    steps = _process_operator(pipeline_tree, operators)
    pipeline_text = 'make_pipeline(\n{STEPS}\n)'.format(STEPS=_indent(',\n'.join(steps), 4))
    return pipeline_text