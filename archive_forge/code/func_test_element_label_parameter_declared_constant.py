from holoviews import element
from holoviews.element import __all__ as all_elements
from holoviews.element.comparison import ComparisonTestCase
def test_element_label_parameter_declared_constant(self):
    """
        Checking all elements in case LabelledData.label is redefined
        """
    for element_name in all_elements:
        el = getattr(element, element_name)
        self.assertEqual(el.param['label'].constant, True, msg=f'Label parameter of element {element_name} not constant')