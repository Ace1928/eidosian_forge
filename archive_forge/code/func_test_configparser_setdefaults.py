from kivy.config import ConfigParser
from os.path import join, dirname
def test_configparser_setdefaults():
    """Test the setdefaults method works as expected."""
    config = ConfigParser()
    config.setdefaults('section', {'test': '1'})
    assert config.get('section', 'test') == '1'