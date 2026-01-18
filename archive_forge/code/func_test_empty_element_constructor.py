import numpy as np
import pandas as pd
import param
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.path import BaseShape
def test_empty_element_constructor(self):
    failed_elements = []
    for name, el in param.concrete_descendents(Element).items():
        if name == 'Sankey':
            continue
        if issubclass(el, (Annotation, BaseShape, Div, Tiles)):
            continue
        try:
            el([])
        except Exception:
            failed_elements.append(name)
    self.assertEqual(failed_elements, [])