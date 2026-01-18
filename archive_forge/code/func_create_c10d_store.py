import datetime
import socket
from contextlib import closing
import torch.distributed as dist
from torch.distributed.elastic.utils.logging import get_logger
def create_c10d_store(is_server: bool, server_addr: str, server_port: int=-1, world_size: int=1, timeout: float=60 * 10, wait_for_workers: bool=True, retries=3):
    if server_port == -1 and world_size > 1:
        raise ValueError(f'server_port must be specified when world_size > 1, got server_port={server_port}, world_size={world_size}')
    if server_port != -1:
        log.info('sever_port: %s, specified, ignoring retries', server_port)
    attempt = retries if server_port == -1 else 1
    while True:
        if server_port != -1:
            port = server_port
        else:
            port = get_free_port()
        log.info('Creating c10d store on %s:%s\n  world_size  : %s\n  is_server   : %s\n  timeout(sec): %s\n', server_addr, port, world_size, is_server, timeout)
        try:
            store = dist.TCPStore(host_name=server_addr, port=port, world_size=world_size, is_master=is_server, timeout=datetime.timedelta(seconds=timeout), wait_for_workers=wait_for_workers)
            if wait_for_workers:
                _check_full_rank(store, world_size)
            log.info('Successfully created c10d store')
            return store
        except RuntimeError as e:
            if str(e) == _ADDRESS_IN_USE:
                if attempt < retries:
                    log.warning('port: %s already in use, attempt: [%s/%s]', port, attempt, retries)
                    attempt += 1
                else:
                    raise RuntimeError(f'on {server_addr}, port: {port} already in use') from e
            else:
                raise