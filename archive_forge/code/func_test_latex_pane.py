from panel.pane import LaTeX
def test_latex_pane(document, comm):
    pane = LaTeX('$\\frac{p^3}{q}$')
    model = pane.get_root(document, comm=comm)
    assert pane._models[model.ref['id']][0] is model
    assert type(model).__name__ == 'KaTeX'
    assert model.text == '$\\frac{p^3}{q}$'
    pane._cleanup(model)
    assert pane._models == {}