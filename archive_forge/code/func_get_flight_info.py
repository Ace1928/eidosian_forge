import asyncio
import pytest
import pyarrow
def get_flight_info(self, context, descriptor):
    if descriptor.command == b'simple':
        return self.simple_info
    elif descriptor.command == b'unknown':
        raise NotImplementedError('Unknown command')
    raise NotImplementedError('Unknown descriptor')