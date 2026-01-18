import pytest
from packaging.version import Version
import numpy as np
import panel as pn
from panel.models.vega import VegaPlot
from panel.pane import PaneBase, Vega
@altair_available
def test_get_vega_pane_type_from_altair():
    assert PaneBase.get_pane_type(altair_example()) is Vega