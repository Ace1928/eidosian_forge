from __future__ import print_function
def get_php_references():
    download = urlretrieve(PHP_MANUAL_URL)
    tar = tarfile.open(download[0])
    tar.extractall()
    tar.close()
    for file in glob.glob('%s%s' % (PHP_MANUAL_DIR, PHP_REFERENCE_GLOB)):
        yield file
    os.remove(download[0])