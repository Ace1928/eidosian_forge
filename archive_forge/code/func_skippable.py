from typing import (
from torch import Tensor, nn
from ..microbatch import Batch
from .namespace import Namespace
from .tracker import current_skip_tracker
def skippable(stash: Iterable[str]=(), pop: Iterable[str]=()) -> Callable[[Type[SkippableModule]], Type[Skippable]]:
    """Define a decorator to create :class:`nn.Module <torch.nn.Module>` with skip connections.

    These decorated modules are called "skippable". This functionality works perfectly
    fine even when the module is not wrapped by :class:`~torch.distributed.pipeline.sync.Pipe`.

    Each skip tensor is managed by its name. Before manipulating skip tensors,
    a skippable module must statically declare the names for skip tensors by
    `stash` and/or `pop` parameters. Skip tensors with pre-declared name can be
    stashed by ``yield stash(name, tensor)`` or popped by ``tensor = yield
    pop(name)``.

    Here is an example with three layers. A skip tensor named "1to3" is stashed
    and popped at the first and last layer, respectively::

        @skippable(stash=['1to3'])
        class Layer1(nn.Module):
            def forward(self, input):
                yield stash('1to3', input)
                return f1(input)

        class Layer2(nn.Module):
            def forward(self, input):
                return f2(input)

        @skippable(pop=['1to3'])
        class Layer3(nn.Module):
            def forward(self, input):
                skip_1to3 = yield pop('1to3')
                return f3(input) + skip_1to3

        model = nn.Sequential(Layer1(), Layer2(), Layer3())

    One skippable module can stash or pop multiple skip tensors::

        @skippable(stash=['alice', 'bob'], pop=['carol'])
        class StashStashPop(nn.Module):
            def forward(self, input):
                yield stash('alice', f_alice(input))
                yield stash('bob', f_bob(input))
                carol = yield pop('carol')
                return input + carol

    Every skip tensor must be associated with exactly one pair of `stash` and
    `pop`. :class:`~torch.distributed.pipeline.sync.Pipe` checks this
    restriction automatically when wrapping a module. You can also check the
    restriction by :func:`verify_skippables`
    without :class:`~torch.distributed.pipeline.sync.Pipe`.

    """
    stashable_names = frozenset(stash)
    poppable_names = frozenset(pop)

    def extend_skippable(module_cls: Type[SkippableModule]) -> Type[Skippable]:
        name = module_cls.__name__
        bases = (Skippable,)
        attrs = {'module_cls': module_cls, 'stashable_names': stashable_names, 'poppable_names': poppable_names}
        return type(name, bases, attrs)
    return extend_skippable