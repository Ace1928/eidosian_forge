from operator import itemgetter
from typing import TYPE_CHECKING, Callable, NamedTuple, Optional, Sequence
from . import errors
from .protocol import is_renderable, rich_cast
def measure_renderables(console: 'Console', options: 'ConsoleOptions', renderables: Sequence['RenderableType']) -> 'Measurement':
    """Get a measurement that would fit a number of renderables.

    Args:
        console (~rich.console.Console): Console instance.
        options (~rich.console.ConsoleOptions): Console options.
        renderables (Iterable[RenderableType]): One or more renderable objects.

    Returns:
        Measurement: Measurement object containing range of character widths required to
            contain all given renderables.
    """
    if not renderables:
        return Measurement(0, 0)
    get_measurement = Measurement.get
    measurements = [get_measurement(console, options, renderable) for renderable in renderables]
    measured_width = Measurement(max(measurements, key=itemgetter(0)).minimum, max(measurements, key=itemgetter(1)).maximum)
    return measured_width