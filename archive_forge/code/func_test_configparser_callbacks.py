from kivy.config import ConfigParser
from os.path import join, dirname
def test_configparser_callbacks():
    """Test that the ConfigParser handles callbacks."""

    def callback():
        pass
    config = ConfigParser()
    assert len(config._callbacks) == 0
    config.add_callback(callback, 'section', 'key')
    assert len(config._callbacks) == 1
    config.remove_callback(callback, 'section', 'key')
    assert len(config._callbacks) == 0