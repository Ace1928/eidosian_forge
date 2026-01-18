def parso_to_jedi_errors(grammar, module_node):
    return [SyntaxError(e) for e in grammar.iter_errors(module_node)]