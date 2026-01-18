from .command import Command
@property
def airplane_mode(self):
    return self.mask % 2 == 1