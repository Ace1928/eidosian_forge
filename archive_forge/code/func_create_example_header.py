from .util import coalesce
def create_example_header():
    """Create an example header with image at top."""
    import wandb.apis.reports as wr
    return [wr.P(), wr.HorizontalRule(), wr.P(), wr.Image('https://camo.githubusercontent.com/83839f20c90facc062330f8fee5a7ab910fdd04b80b4c4c7e89d6d8137543540/68747470733a2f2f692e696d6775722e636f6d2f676236423469672e706e67'), wr.P(), wr.HorizontalRule(), wr.P()]