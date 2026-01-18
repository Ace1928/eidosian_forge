from __future__ import annotations
import functools
import json
import sys
import typing
import click
import httpcore
import pygments.lexers
import pygments.util
import rich.console
import rich.markup
import rich.progress
import rich.syntax
import rich.table
from ._client import Client
from ._exceptions import RequestError
from ._models import Response
from ._status_codes import codes
def download_response(response: Response, download: typing.BinaryIO) -> None:
    console = rich.console.Console()
    console.print()
    content_length = response.headers.get('Content-Length')
    with rich.progress.Progress('[progress.description]{task.description}', '[progress.percentage]{task.percentage:>3.0f}%', rich.progress.BarColumn(bar_width=None), rich.progress.DownloadColumn(), rich.progress.TransferSpeedColumn()) as progress:
        description = f'Downloading [bold]{rich.markup.escape(download.name)}'
        download_task = progress.add_task(description, total=int(content_length or 0), start=content_length is not None)
        for chunk in response.iter_bytes():
            download.write(chunk)
            progress.update(download_task, completed=response.num_bytes_downloaded)