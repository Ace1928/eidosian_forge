import warnings
from io import StringIO
import time
from urllib.parse import urlencode
from urllib.request import build_opener, install_opener
from urllib.request import urlopen
from urllib.request import HTTPPasswordMgrWithDefaultRealm, HTTPBasicAuthHandler
from urllib.request import Request
from Bio import BiopythonWarning
from Bio._utils import function_with_previous
@function_with_previous
def qblast(program, database, sequence, url_base=NCBI_BLAST_URL, auto_format=None, composition_based_statistics=None, db_genetic_code=None, endpoints=None, entrez_query='(none)', expect=10.0, filter=None, gapcosts=None, genetic_code=None, hitlist_size=50, i_thresh=None, layout=None, lcase_mask=None, matrix_name=None, nucl_penalty=None, nucl_reward=None, other_advanced=None, perc_ident=None, phi_pattern=None, query_file=None, query_believe_defline=None, query_from=None, query_to=None, searchsp_eff=None, service=None, threshold=None, ungapped_alignment=None, word_size=None, short_query=None, alignments=500, alignment_view=None, descriptions=500, entrez_links_new_window=None, expect_low=None, expect_high=None, format_entrez_query=None, format_object=None, format_type='XML', ncbi_gi=None, results_file=None, show_overview=None, megablast=None, template_type=None, template_length=None, username='blast', password=None):
    """BLAST search using NCBI's QBLAST server or a cloud service provider.

    Supports all parameters of the old qblast API for Put and Get.

    Please note that NCBI uses the new Common URL API for BLAST searches
    on the internet (http://ncbi.github.io/blast-cloud/dev/api.html). Thus,
    some of the parameters used by this function are not (or are no longer)
    officially supported by NCBI. Although they are still functioning, this
    may change in the future.

    The Common URL API (http://ncbi.github.io/blast-cloud/dev/api.html) allows
    doing BLAST searches on cloud servers. To use this feature, please set
    ``url_base='http://host.my.cloud.service.provider.com/cgi-bin/blast.cgi'``
    and ``format_object='Alignment'``. For more details, please see
    https://blast.ncbi.nlm.nih.gov/Blast.cgi?PAGE_TYPE=BlastDocs&DOC_TYPE=CloudBlast

    Some useful parameters:

     - program        blastn, blastp, blastx, tblastn, or tblastx (lower case)
     - database       Which database to search against (e.g. "nr").
     - sequence       The sequence to search.
     - ncbi_gi        TRUE/FALSE whether to give 'gi' identifier.
     - descriptions   Number of descriptions to show.  Def 500.
     - alignments     Number of alignments to show.  Def 500.
     - expect         An expect value cutoff.  Def 10.0.
     - matrix_name    Specify an alt. matrix (PAM30, PAM70, BLOSUM80, BLOSUM45).
     - filter         "none" turns off filtering.  Default no filtering
     - format_type    "HTML", "Text", "ASN.1", or "XML".  Def. "XML".
     - entrez_query   Entrez query to limit Blast search
     - hitlist_size   Number of hits to return. Default 50
     - megablast      TRUE/FALSE whether to use MEga BLAST algorithm (blastn only)
     - short_query    TRUE/FALSE whether to adjust the search parameters for a
                      short query sequence. Note that this will override
                      manually set parameters like word size and e value. Turns
                      off when sequence length is > 30 residues. Default: None.
     - service        plain, psi, phi, rpsblast, megablast (lower case)

    This function does no checking of the validity of the parameters
    and passes the values to the server as is.  More help is available at:
    https://ncbi.github.io/blast-cloud/dev/api.html

    """
    programs = ['blastn', 'blastp', 'blastx', 'tblastn', 'tblastx']
    if program not in programs:
        raise ValueError(f'Program specified is {program}. Expected one of {', '.join(programs)}')
    if short_query and program == 'blastn':
        short_query = None
        if len(sequence) < 31:
            expect = 1000
            word_size = 7
            nucl_reward = 1
            filter = None
            lcase_mask = None
            warnings.warn('"SHORT_QUERY_ADJUST" is incorrectly implemented (by NCBI) for blastn. We bypass the problem by manually adjusting the search parameters. Thus, results may slightly differ from web page searches.', BiopythonWarning)
    parameters = {'AUTO_FORMAT': auto_format, 'COMPOSITION_BASED_STATISTICS': composition_based_statistics, 'DATABASE': database, 'DB_GENETIC_CODE': db_genetic_code, 'ENDPOINTS': endpoints, 'ENTREZ_QUERY': entrez_query, 'EXPECT': expect, 'FILTER': filter, 'GAPCOSTS': gapcosts, 'GENETIC_CODE': genetic_code, 'HITLIST_SIZE': hitlist_size, 'I_THRESH': i_thresh, 'LAYOUT': layout, 'LCASE_MASK': lcase_mask, 'MEGABLAST': megablast, 'MATRIX_NAME': matrix_name, 'NUCL_PENALTY': nucl_penalty, 'NUCL_REWARD': nucl_reward, 'OTHER_ADVANCED': other_advanced, 'PERC_IDENT': perc_ident, 'PHI_PATTERN': phi_pattern, 'PROGRAM': program, 'QUERY': sequence, 'QUERY_FILE': query_file, 'QUERY_BELIEVE_DEFLINE': query_believe_defline, 'QUERY_FROM': query_from, 'QUERY_TO': query_to, 'SEARCHSP_EFF': searchsp_eff, 'SERVICE': service, 'SHORT_QUERY_ADJUST': short_query, 'TEMPLATE_TYPE': template_type, 'TEMPLATE_LENGTH': template_length, 'THRESHOLD': threshold, 'UNGAPPED_ALIGNMENT': ungapped_alignment, 'WORD_SIZE': word_size, 'CMD': 'Put'}
    if password is not None:
        password_mgr = HTTPPasswordMgrWithDefaultRealm()
        password_mgr.add_password(None, url_base, username, password)
        handler = HTTPBasicAuthHandler(password_mgr)
        opener = build_opener(handler)
        install_opener(opener)
    if url_base == NCBI_BLAST_URL:
        parameters.update({'email': email, 'tool': tool})
    parameters = {key: value for key, value in parameters.items() if value is not None}
    message = urlencode(parameters).encode()
    request = Request(url_base, message, {'User-Agent': 'BiopythonClient'})
    handle = urlopen(request)
    rid, rtoe = _parse_qblast_ref_page(handle)
    parameters = {'ALIGNMENTS': alignments, 'ALIGNMENT_VIEW': alignment_view, 'DESCRIPTIONS': descriptions, 'ENTREZ_LINKS_NEW_WINDOW': entrez_links_new_window, 'EXPECT_LOW': expect_low, 'EXPECT_HIGH': expect_high, 'FORMAT_ENTREZ_QUERY': format_entrez_query, 'FORMAT_OBJECT': format_object, 'FORMAT_TYPE': format_type, 'NCBI_GI': ncbi_gi, 'RID': rid, 'RESULTS_FILE': results_file, 'SERVICE': service, 'SHOW_OVERVIEW': show_overview, 'CMD': 'Get'}
    parameters = {key: value for key, value in parameters.items() if value is not None}
    message = urlencode(parameters).encode()
    delay = 20
    while True:
        current = time.time()
        wait = qblast.previous + delay - current
        if wait > 0:
            time.sleep(wait)
            qblast.previous = current + wait
        else:
            qblast.previous = current
        if delay < 60 and url_base == NCBI_BLAST_URL:
            delay = 60
        request = Request(url_base, message, {'User-Agent': 'BiopythonClient'})
        handle = urlopen(request)
        results = handle.read().decode()
        if results == '\n\n':
            continue
        if 'Status=' not in results:
            break
        i = results.index('Status=')
        j = results.index('\n', i)
        status = results[i + len('Status='):j].strip()
        if status.upper() == 'READY':
            break
    return StringIO(results)