from reportlab.graphics.shapes import *
from reportlab.lib.validators import DerivedValue
from reportlab import rl_config
from . transform import mmult, inverse
def testStateTracker():
    print('Testing state tracker')
    defaults = {'fillColor': None, 'strokeColor': None, 'fontName': None, 'transform': [1, 0, 0, 1, 0, 0]}
    from reportlab.graphics.shapes import _baseGFontName
    deltas = [{'fillColor': 'red'}, {'fillColor': 'green', 'strokeColor': 'blue', 'fontName': _baseGFontName}, {'transform': [0.5, 0, 0, 0.5, 0, 0]}, {'transform': [0.5, 0, 0, 0.5, 2, 3]}, {'strokeColor': 'red'}]
    st = StateTracker(defaults)
    print('initial:', st.getState())
    print()
    for delta in deltas:
        print('pushing:', delta)
        st.push(delta)
        print('state:  ', st.getState(), '\n')
    for delta in deltas:
        print('popping:', st.pop())
        print('state:  ', st.getState(), '\n')