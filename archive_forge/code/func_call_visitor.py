from sympy.external import import_module
def call_visitor(fort_node):
    """Calls the AST Visitor on the Module

    This function is used to call the AST visitor for a program or module
    It imports all the required modules and calls the visit() function
    on the given node

    Parameters
    ==========

    fort_node : LFortran ASR object
        Node for the operation for which the NodeVisitor is called

    Returns
    =======

    res_ast : list
        list of SymPy AST Nodes

    """
    v = ASR2PyVisitor()
    v.visit(fort_node)
    res_ast = v.ret_ast()
    return res_ast