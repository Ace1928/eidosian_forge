import logging
import os
import stat
import sys
from pyomo.common.download import FileDownloader
def get_gjh(downloader):
    system, bits = downloader.get_sysinfo()
    url = downloader.get_platform_url(urlmap)
    downloader.set_destination_filename(os.path.join('bin', 'gjh' + exemap[system]))
    logger.info('Fetching GJH from %s and installing it to %s' % (url, downloader.destination()))
    downloader.get_gzipped_binary_file(url)
    mode = os.stat(downloader.destination()).st_mode
    os.chmod(downloader.destination(), mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)