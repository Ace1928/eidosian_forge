import weakref, ctypes, logging, os, glob
from OpenGL.platform import ctypesloader
from OpenGL import _opaque
def filter_bad_drivers(cards, bad_drivers=('nvidia',)):
    """Lookup the driver for each card to exclude loading nvidia devices"""
    bad_cards = set()
    for link in glob.glob('/dev/dri/by-path/pci-*-card'):
        base = os.path.basename(link)
        pci_id = base[4:-5]
        driver = os.path.basename(os.readlink('/sys/bus/pci/devices/%s/driver' % (pci_id,)))
        if driver in bad_drivers:
            card = os.path.basename(os.readlink(link))
            log.debug('Skipping %s because it uses %s driver', card, driver)
            bad_cards.add(card)
    filtered = [card for card in cards if os.path.basename(card) not in bad_cards]
    return filtered