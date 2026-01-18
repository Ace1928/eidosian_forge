from .util import coalesce
def create_example_footer():
    """Create an example footer with image and text at bottom."""
    import wandb.apis.reports as wr
    return [wr.P(), wr.HorizontalRule(), wr.P(), wr.H1('Disclaimer'), wr.P('The views and opinions expressed in this report are those of the authors and do not necessarily reflect the official policy or position of Weights & Biases. blah blah blah blah blah boring text at the bottom'), wr.P(), wr.HorizontalRule()]