import pytest
from string import ascii_letters
from random import randint
import gc
import sys
@pytest.fixture
def kivy_benchmark(benchmark, kivy_clock):
    from kivy.core.window import Window
    from kivy.cache import Cache
    from kivy.utils import platform
    import kivy
    from kivy.core.gl import glGetString, GL_VENDOR, GL_RENDERER, GL_VERSION
    from kivy.context import Context
    from kivy.clock import ClockBase
    from kivy.factory import FactoryBase, Factory
    from kivy.lang.builder import BuilderBase, Builder
    context = Context(init=False)
    context['Clock'] = ClockBase()
    context['Factory'] = FactoryBase.create_from(Factory)
    context['Builder'] = BuilderBase.create_from(Builder)
    for category in list(Cache._objects.keys()):
        if category not in Cache._categories:
            continue
        for key in list(Cache._objects[category].keys()):
            Cache.remove(category, key)
    gc.collect()
    benchmark.extra_info['platform'] = str(sys.platform)
    benchmark.extra_info['python_version'] = str(sys.version)
    benchmark.extra_info['python_api'] = str(sys.api_version)
    benchmark.extra_info['kivy_platform'] = platform
    benchmark.extra_info['kivy_version'] = kivy.__version__
    benchmark.extra_info['gl_vendor'] = str(glGetString(GL_VENDOR))
    benchmark.extra_info['gl_renderer'] = str(glGetString(GL_RENDERER))
    benchmark.extra_info['gl_version'] = str(glGetString(GL_VERSION))
    context.push()
    try:
        yield benchmark
    finally:
        context.pop()