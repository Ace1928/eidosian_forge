import deap
from copy import copy
def generate_export_pipeline_code(pipeline_tree, operators):
    """Generate code specific to the construction of the sklearn Pipeline for export_pipeline.

    Parameters
    ----------
    pipeline_tree: list
        List of operators in the current optimized pipeline

    Returns
    -------
    Source code for the sklearn pipeline

    """
    steps = _process_operator(pipeline_tree, operators)
    num_step = len(steps)
    if num_step > 1:
        pipeline_text = 'make_pipeline(\n{STEPS}\n)'.format(STEPS=_indent(',\n'.join(steps), 4))
    else:
        pipeline_text = '{STEPS}'.format(STEPS=_indent(',\n'.join(steps), 0))
    return pipeline_text