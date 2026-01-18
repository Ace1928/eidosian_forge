import json
import logging
import sys
import threading
import time
from typing import Optional
import etcd  # type: ignore[import]
from torch.distributed.elastic.rendezvous import (
from .utils import parse_rendezvous_endpoint
from .etcd_store import EtcdStore, cas_delay
def rendezvous_barrier(self):
    """
        Main entry point for next rendezvous.

        This method is blocking until rendezvous succeeds or a timeout occurs.

        Returns:
             ``(rdzv_version, rank, world_size)``

        Raises:
            RendezvousTimeoutError - timeout waiting for rendezvous
            RendezvousClosedError - rendezvous is or was closed while waiting
            RendezvousError - other persistent errors that
             render the rendezvous non-retryable
        """
    self._rendezvous_deadline = time.time() + self._timeout
    while True:
        if time.time() > self._rendezvous_deadline:
            raise RendezvousTimeoutError()
        log.info('Attempting to join next rendezvous')
        try:
            if self._lease_this_rank_stop is not None:
                self._lease_this_rank_stop.set()
            return self.init_phase()
        except EtcdRendezvousRetryImmediately:
            pass
        except EtcdRendezvousRetryableFailure:
            time.sleep(1)
        except RendezvousTimeoutError:
            log.info('Rendezvous timeout occurred in EtcdRendezvousHandler')
            raise
        except RendezvousClosedError:
            log.info('Rendezvous for run_id=%s was observed to be closed', self._run_id)
            raise
        except RendezvousError:
            raise
        except Exception as e:
            log.info('Rendezvous attempt failed, will retry. Reason: %s', e)
            time.sleep(1)