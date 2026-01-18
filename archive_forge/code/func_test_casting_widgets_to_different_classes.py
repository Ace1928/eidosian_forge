from unittest import TestCase, SkipTest
from hvplot.util import process_xarray  # noqa
def test_casting_widgets_to_different_classes(self):
    import panel as pn
    pane = self.flowers.hvplot.scatter(groupby='species', legend='top_right', widgets={'species': pn.widgets.DiscreteSlider})
    assert len(look_for_class(pane, pn.widgets.DiscreteSlider)) == 1