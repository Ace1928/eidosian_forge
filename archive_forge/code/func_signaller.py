from typing import Generator
from simpy.core import Environment, SimTime
from simpy.events import Event, Process, ProcessGenerator
def signaller(signaller: Event, receiver: Process) -> ProcessGenerator:
    result = (yield signaller)
    if receiver.is_alive:
        receiver.interrupt((signaller, result))