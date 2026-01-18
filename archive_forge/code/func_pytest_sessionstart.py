import pytest
from hypothesis import settings
def pytest_sessionstart(session):
    try:
        import tensorflow as tf
        physical_devices = tf.config.list_physical_devices('GPU')
        for device in physical_devices:
            try:
                tf.config.experimental.set_memory_growth(device, True)
            except:
                print(f'failed to enable Tensorflow memory growth on {device}')
    except ImportError:
        pass