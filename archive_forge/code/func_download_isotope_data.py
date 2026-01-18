from urllib import request
def download_isotope_data():
    """Download isotope data from NIST public website.

    Relative atomic masses of individual isotopes their abundance
    (mole fraction) are compiled into a dictionary. Individual items can be
    indexed by the atomic number and mass number, e.g. titanium-48:

    >>> from ase.data.isotopes import download_isotope_data
    >>> isotopes = download_isotope_data()
    >>> isotopes[22][48]['mass']
    47.94794198
    >>> isotopes[22][48]['composition']
    0.7372
    """
    url = 'http://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=&ascii=ascii&isotype=all'
    with request.urlopen(url) as fd:
        txt = fd.read()
    raw_data = txt.decode().splitlines()
    return parse_isotope_data(raw_data)