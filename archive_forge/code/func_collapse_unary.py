from nltk.tree.tree import Tree
def collapse_unary(tree, collapsePOS=False, collapseRoot=False, joinChar='+'):
    """
    Collapse subtrees with a single child (ie. unary productions)
    into a new non-terminal (Tree node) joined by 'joinChar'.
    This is useful when working with algorithms that do not allow
    unary productions, and completely removing the unary productions
    would require loss of useful information.  The Tree is modified
    directly (since it is passed by reference) and no value is returned.

    :param tree: The Tree to be collapsed
    :type  tree: Tree
    :param collapsePOS: 'False' (default) will not collapse the parent of leaf nodes (ie.
                        Part-of-Speech tags) since they are always unary productions
    :type  collapsePOS: bool
    :param collapseRoot: 'False' (default) will not modify the root production
                         if it is unary.  For the Penn WSJ treebank corpus, this corresponds
                         to the TOP -> productions.
    :type collapseRoot: bool
    :param joinChar: A string used to connect collapsed node values (default = "+")
    :type  joinChar: str
    """
    if collapseRoot == False and isinstance(tree, Tree) and (len(tree) == 1):
        nodeList = [tree[0]]
    else:
        nodeList = [tree]
    while nodeList != []:
        node = nodeList.pop()
        if isinstance(node, Tree):
            if len(node) == 1 and isinstance(node[0], Tree) and (collapsePOS == True or isinstance(node[0, 0], Tree)):
                node.set_label(node.label() + joinChar + node[0].label())
                node[0:] = [child for child in node[0]]
                nodeList.append(node)
            else:
                for child in node:
                    nodeList.append(child)