import deap
from copy import copy
def expr_to_tree(ind, pset):
    """Convert the unstructured DEAP pipeline into a tree data-structure.

    Parameters
    ----------
    ind: deap.creator.Individual
       The pipeline that is being exported

    Returns
    -------
    pipeline_tree: list
       List of operators in the current optimized pipeline

    EXAMPLE:
        pipeline:
            "DecisionTreeClassifier(input_matrix, 28.0)"
        pipeline_tree:
            ['DecisionTreeClassifier', 'input_matrix', 28.0]

    """

    def prim_to_list(prim, args):
        if isinstance(prim, deap.gp.Terminal):
            if prim.name in pset.context:
                return pset.context[prim.name]
            else:
                return prim.value
        return [prim.name] + args
    tree = []
    stack = []
    for node in ind:
        stack.append((node, []))
        while len(stack[-1][1]) == stack[-1][0].arity:
            prim, args = stack.pop()
            tree = prim_to_list(prim, args)
            if len(stack) == 0:
                break
            stack[-1][1].append(tree)
    return tree