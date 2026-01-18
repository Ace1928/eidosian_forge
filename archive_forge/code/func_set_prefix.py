from .trait_base import class_of
def set_prefix(self, prefix):
    if hasattr(self, 'prefix'):
        self.prefix = prefix
        self.set_args()